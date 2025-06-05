from itertools import chain
from comfy.text_encoders.flux import FluxClipModel
from comfy.text_encoders.sd3_clip import SD3ClipModel
from comfy.sd1_clip import escape_important, token_weights


class PromptAnalyzer:
    def __init__(self, clip, all_tokens, model=None):
        self.clip = clip
        self.tokens = all_tokens
        self.index_offset = 0

        if model is None:
            if isinstance(clip.cond_stage_model, FluxClipModel):
                self.model = "t5xxl"
            elif isinstance(clip.cond_stage_model, SD3ClipModel):
                self.model = "t5xxl"
                # SD3 embeds both CLIP and T5 in the token list. We take T5 indices by skipping CLIP tokens.
                self.index_offset = len(self._get_tokens_list(all_tokens, "l"))
            else:
                # SDXL & SD1.5
                self.model = "l"
        else:
            self.model = model

        self.end_token = self._get_end_token()

    def _get_end_token(self):
        if self.model == "t5xxl":
            return 1  # </s>
        else:
            # SDXL & SD1.5
            return self.clip.cond_stage_model.clip_l.special_tokens["end"]

    def _get_tokens_list(self, tokens, model=None):
        if model is None:
            model = self.model
        return list(chain.from_iterable(tokens[model]))

    def encode(self, text: str):
        tokens = self.clip.tokenize(text)
        tokens = self._get_tokens_list(tokens)

        if self.model == "t5xxl":
            start_idx = 0
        else:
            start_idx = 1

        for i in range(len(tokens)):
            if tokens[i][0] == self.end_token:
                return tokens[start_idx:i]

    def calc_word_indices(self, word: str, limit: int = -1, start_pos=0):
        if self.model == "t5xxl":
            return self._calc_word_indices_t5xxl(word)
        else:
            return self._calc_word_indices_clip(word, limit, start_pos)

    def _calc_word_indices_clip(self, word: str, limit: int = -1, start_pos=0):
        word = word.lower()

        # Remove weight specifiers from the query but keep the escaped parentheses
        cleaned_word, _ = token_weights(escape_important(word), 1.0)[0]
        cleaned_word = cleaned_word.replace("\0\1", "\\)")
        cleaned_word = cleaned_word.replace("\0\2", "\\(")
        is_specifier_removed = cleaned_word != word

        # Handle special case where a comma should be included at the end of the word
        # curl -s https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/tokenizer/vocab.json | grep ,\</w\>
        special_case = (
            "!)",
            "!",
            '"',
            "%)",
            "%",
            "'",
            ")",
            "*",
            "+",
            ".",
            "..",
            "...",
            "?",
            "]",
            "_",
            "âģ©",
        )

        if not is_specifier_removed and cleaned_word.endswith(special_case):
            cleaned_word += ","

        merge_idxs = []

        tokens = self.tokens
        tokens = self._get_tokens_list(tokens)
        needles = self.encode(cleaned_word)

        limit_count = 0

        if len(needles) == 0:
            return merge_idxs, start_pos

        for i, token in enumerate(tokens):
            if i < start_pos:
                continue

            if needles[0][0] == token[0] and len(needles) > 1:
                next = i + 1
                success = True
                for needle in needles[1:]:
                    if next >= len(tokens) or needle[0] != tokens[next][0]:
                        success = False
                        break
                    next += 1

                # append consecutive indexes if all pass
                if success:
                    merge_idxs.extend(list(range(i, next)))
                    if limit > 0:
                        limit_count += 1
                        if limit_count >= limit:
                            break

            elif needles[0][0] == token[0]:
                merge_idxs.append(i)
                if limit > 0:
                    limit_count += 1
                    if limit_count >= limit:
                        break

        return merge_idxs

    def _calc_word_indices_t5xxl(self, word: str):
        word = word.lower()
        tokens = self.tokens
        tokens = self._get_tokens_list(tokens)
        detok_tokens = self.clip.tokenizer.t5xxl.untokenize(tokens)

        needles = self.encode(word)
        detok_query = self.clip.tokenizer.t5xxl.untokenize(needles)

        # Extract just the token text from both arrays, ignoring padding and end tokens
        token_texts = [
            token[1] for token in detok_tokens if token[1] not in ["<pad>", "</s>"]
        ]
        query_texts = [
            token[1] for token in detok_query if token[1] not in ["<pad>", "</s>"]
        ]

        # Clean tokens by removing '▁' character
        clean_token_texts = [text.replace("▁", "") for text in token_texts]
        clean_query_texts = [text.replace("▁", "") for text in query_texts]

        # Combined query text for matching (without the '▁' characters)
        joined_query = "".join(clean_query_texts)

        matches = []
        for i in range(len(clean_token_texts)):
            # Try to match the entire query starting at this position
            current_combined = ""
            for j in range(i, len(clean_token_texts)):
                current_combined += clean_token_texts[j]

                # Check if we've found the complete query
                if current_combined == joined_query:
                    # Return indices from i to j (inclusive)
                    matches = list(
                        range(i + self.index_offset, j + self.index_offset + 1)
                    )
                    return matches

                # If we've accumulated more characters than the query, this starting position won't work
                if len(current_combined) > len(joined_query):
                    break

        return matches
