import itertools
import random

class PromptManager:
    def __init__(self, initial_caption: str):
        self.caption = initial_caption
        # Define prompt template and blocks (factors for DOE)
        # Start with the caption as the base prompt
        self.template = "{caption}{style}{lighting}"
        # Define some example blocks with default options
        self.blocks = {
            "caption": [initial_caption],  # the base description (fixed)
            "style": ["", ", in a photograph", ", digital art"],  # style variations
            "lighting": ["", ", golden hour lighting", ", night time"]  # lighting vars
        }
        # Negative prompt components can also be managed here
        self.negative_base = "low quality, blurry"
        self.negative_extra = []  # additional negatives from observations

    def set_block_options(self, block_name: str, options: list):
        """Manually set the options for a given prompt block factor."""
        if block_name in self.blocks:
            self.blocks[block_name] = options
        else:
            self.blocks[block_name] = options

    def generate_prompt_variants(self, max_combinations: int = None):
        """
        Generate a list of prompt strings by combining blocks.
        If max_combinations is set, randomly sample that many combinations (if combinations are more).
        """
        # All combinations of one choice from each block:
        keys = list(self.blocks.keys())
        all_combos = list(itertools.product(*[self.blocks[k] for k in keys]))
        if max_combinations and len(all_combos) > max_combinations:
            random.seed(0)  # for reproducibility, though this could be configurable
            random.shuffle(all_combos)
            combos = all_combos[:max_combinations]
        else:
            combos = all_combos
        prompts = []
        for combo in combos:
            combo_dict = {key: (val if val != "" else "") for key, val in zip(keys, combo)}
            prompt = self.template.format(**combo_dict).strip()
            # Remove double spaces or trailing commas due to empty blocks
            prompt = prompt.replace("  ", " ").replace(" ,", ",").strip().strip(",")
            prompts.append(prompt)
        return prompts

    def update_from_scores(self, prompt_scores: dict):
        """
        Simple heuristic: pick the best scoring prompt and use its blocks for next iteration.
        prompt_scores: dict mapping prompt text -> score.
        """
        if not prompt_scores:
            return
        best_prompt = max(prompt_scores, key=lambda p: prompt_scores[p])
        # For each block, if the best_prompt contains one of the block options, keep that option.
        for block, options in self.blocks.items():
            for opt in options:
                if opt and opt in best_prompt:
                    self.blocks[block] = [opt]  # narrow to this best option
                    break
        # (In a more sophisticated approach, we could also adjust negative prompts or add new factors based on differences.)

    def get_negative_prompt(self):
        """Assemble the current negative prompt string."""
        parts = [self.negative_base] + self.negative_extra
        neg = ", ".join([p for p in parts if p])
        return neg
