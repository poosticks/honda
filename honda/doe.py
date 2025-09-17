import itertools
import math
import random

class DOEModule:
    def __init__(self, seeds:list):
        """
        seeds: a list of random seeds to use for image generation. 
        We use a fixed seed list for all experiments for consistency.
        """
        self.seeds = seeds

    def plan_experiments(self, prompts:list, max_images:int=None):
        """
        Given a list of prompt variants and a pool of seeds, decide which (prompt, seed) pairs to run.
        If max_images is specified, cap the total number of images.
        Returns a list of (prompt, seed) tuples.
        """
        pairs = [(p, s) for p in prompts for s in self.seeds]
        total = len(pairs)
        if max_images and total > max_images:
            # If too many, sample evenly across prompts.
            # E.g., take at most ceil(max_images/len(prompts)) seeds per prompt.
            num_prompts = len(prompts)
            max_per_prompt = math.ceil(max_images / num_prompts)
            new_pairs = []
            for p in prompts:
                # take the first max_per_prompt seeds for each prompt
                for s in self.seeds[:max_per_prompt]:
                    new_pairs.append((p, s))
            pairs = new_pairs[:max_images]
        return pairs

    def next_seed_subset(self, num:int):
        """
        Optionally, if varying seeds per iteration: pick a subset of seeds for the next iteration.
        Not used in simple implementation (we keep fixed seeds).
        """
        if num >= len(self.seeds):
            return self.seeds
        random.seed(0)
        return random.sample(self.seeds, num)
