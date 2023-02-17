"""
Greedy Word Swap with Word Importance Ranking
===================================================


When WIR method is set to ``unk``, this is a reimplementation of the search
method from the paper: Is BERT Really Robust?

A Strong Baseline for Natural Language Attack on Text Classification and
Entailment by Jin et. al, 2019. See https://arxiv.org/abs/1907.11932 and
https://github.com/jind11/TextFooler.
"""


from torch.nn.functional import softmax
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)
from abc import ABC, abstractmethod
from collections import defaultdict
import os
import numpy as np
import torch
from textattack.shared import utils
from textattack.shared import AttackedText
import operator

class GreedyWordSwapWIR(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """

    def __init__(self, wir_method="unk"):
        self.wir_method = wir_method

    def normalize(self,n):
        n = np.array(n)
        n[n < 0] = 0
        s = np.sum(n)
        if s == 0:
            return np.ones(len(n)) / len(n)
        else:
            return n / s


    def _get_best_neighbors(self, current_result, original_result):
        """For given current text, find its neighboring texts that yields
        maximum improvement (in goal function score) for each word.

        Args:
            current_result (GoalFunctionResult): `GoalFunctionResult` of current text
            original_result (GoalFunctionResult): `GoalFunctionResult` of original text.
        Returns:
            best_neighbors (list[GoalFunctionResult]): Best neighboring text for each word
            prob_list (list[float]): discrete probablity distribution for sampling a neighbor from `best_neighbors`
        """
        current_text = current_result.attacked_text
        neighbors_list = [[] for _ in range(len(current_text.words))]
        transformed_texts = self.get_transformations(
            current_text, original_text=original_result.attacked_text
        )
        for transformed_text in transformed_texts:
            diff_idx = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
            neighbors_list[diff_idx].append(transformed_text)

        best_neighbors = []
        score_list = []
        for i in range(len(neighbors_list)):
            if not neighbors_list[i]:
                best_neighbors.append(current_result)
                score_list.append(0)
                continue

            neighbor_results, self._search_over = self.get_goal_results(
                neighbors_list[i]
            )
            if not len(neighbor_results):
                best_neighbors.append(current_result)
                score_list.append(0)
            else:
                neighbor_scores = np.array([r.score for r in neighbor_results])
                score_diff = neighbor_scores - current_result.score

                best_idx = np.argmax(neighbor_scores)
                best_neighbors.append(neighbor_results[best_idx])
                score_list.append(score_diff[best_idx])

        prob_list = normalize(score_list)

        return best_neighbors, prob_list


    def _perturb(self, pop_member, original_result):
        """Perturb `pop_member` in-place.
        Replaces a word at a random in `pop_member` with replacement word that maximizes increase in score.
        Args:
            pop_member (PopulationMember): The population member being perturbed.
            original_result (GoalFunctionResult): Result of original sample being attacked
        Returns:
            `True` if perturbation occured. `False` if not.
        """
        # TODO: Below is very slow and is the main cause behind memory build up + slowness
        best_neighbors, prob_list = self._get_best_neighbors(
            pop_member, original_result
        )
        random_result = np.random.choice(best_neighbors, 1, p=prob_list)[0]
        pop_member.attacked_text = random_result.attacked_text
        pop_member = random_result

        return pop_member


    def _get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""
        len_text = len(initial_text.words)

        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "weighted-saliency":
            # first, compute word saliency
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            saliency_scores = np.array([result.score for result in leave_one_results])

            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()

            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in range(len_text):
                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, _ = self.get_goal_results(transformed_text_candidates)
                score_change = [result.score for result in swap_results]
                if not score_change:
                    delta_ps.append(0.0)
                    continue
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = softmax_saliency_scores

        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "gradient":

            victim_model = self.get_victim_model()
            gradient_scores = np.zeros(initial_text.num_words)
            grad_output = victim_model.get_grad(initial_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = initial_text.align_with_model_tokens(victim_model)
            for i, word in enumerate(initial_text.words):
                matched_tokens = word2token_mapping[i]
                if not matched_tokens:
                    gradient_scores[i] = 0.0
                else:
                    ave_grad = np.mean(gradient[matched_tokens], axis=0)
                    gradient_scores[i] = np.linalg.norm(ave_grad, ord=2)

            index_scores = gradient_scores
            search_over = False

        elif self.wir_method == "random":
            index_order = np.arange(len_text)
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        if self.wir_method != "random":
            index_order = (-index_scores).argsort()

        return index_order, search_over

    def perform_search(self, initial_result):

        attacked_text = initial_result.attacked_text
        # Sort words by order of importance

        index_order, search_over = self._get_index_order(attacked_text)

        i = 0
        cur_result = initial_result
        results = None
        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                continue

            change_ratio = initial_result.attacked_text.words_diff_ratio(
                cur_result.attacked_text
            )
            p_change = 1 - 2 * change_ratio
            if np.random.uniform() < p_change:
                cur_result = self._perturb(cur_result, initial_result)
                cur_result, search_over = self.get_goal_results([cur_result.attacked_text])
                cur_result = cur_result[0]
            # If we succeeded, return the index with best similarity.
            index_order_1, search_over_1 = self._get_index_order(cur_result.attacked_text)

            if index_order_1.all() == index_order.all():
                pass
            else:
                index_order = index_order_1
                i = 1

            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return best_result
        return cur_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        if self.wir_method == "gradient":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]

def normalize(n):
    n = np.array(n)
    n[n < 0] = 0
    s = np.sum(n)
    if s == 0:
        return np.ones(len(n)) / len(n)
    else:
        return n / s
