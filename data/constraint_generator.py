import random
import spacy
import openai
from typing import List, Dict, Tuple, Optional
from ..utils.constraint_utils import ConstraintProcessor


class ConstraintDataGenerator:

    def __init__(self, openai_api_key: Optional[str] = None):
        self.nlp = spacy.load("en_core_web_sm")
        self.constraint_processor = ConstraintProcessor()

        # Initialize OpenAI client for GPT-4o sentiment labeling
        if openai_api_key:
            openai.api_key = openai_api_key
        self.openai_client = openai

    def extract_pos_pattern_from_input(self, input_text: str) -> str:
        
        doc = self.nlp(input_text)
        pos_tags = []

        for token in doc:
            if not token.is_space and not token.is_punct:
                # Map spaCy POS tags to more standard tags
                if token.pos_ == "DET":
                    pos_tags.append("DT")
                elif token.pos_ == "NOUN":
                    pos_tags.append("NOUN")
                elif token.pos_ == "VERB":
                    pos_tags.append("VERB")
                elif token.pos_ == "ADJ":
                    pos_tags.append("ADJ")
                elif token.pos_ == "ADV":
                    pos_tags.append("ADV")
                elif token.pos_ == "PRON":
                    pos_tags.append("PRON")
                elif token.pos_ == "ADP":
                    pos_tags.append("PREP")
                elif token.pos_ == "CONJ":
                    pos_tags.append("CONJ")
                elif token.pos_ == "NUM":
                    pos_tags.append("NUM")
                else:
                    pos_tags.append(token.pos_)

        return " ".join(pos_tags)

    def extract_syntactic_structure_from_input(self, input_text: str) -> str:

        doc = self.nlp(input_text)

        def token_to_pos_tag(token):

            pos_mapping = {
                "DET": "DT",
                "NOUN": "NN",
                "VERB": "VBD" if token.tag_ in ["VBD"] else "VBZ" if token.tag_ in ["VBZ"] else "VB",
                "ADJ": "JJ",
                "ADV": "RB",
                "PRON": "PRP",
                "ADP": "IN",
                "CONJ": "CC",
                "NUM": "CD",
                "PROPN": "NNP"
            }
            return pos_mapping.get(token.pos_, token.tag_)

        def build_tree_structure(sent):
            """Build a simplified parse tree structure"""
            # Get the root token
            root = None
            for token in sent:
                if token.head == token:
                    root = token
                    break

            if not root:
                return ""

            def get_children_by_dep(token, dep_label):
                return [child for child in token.children if child.dep_ == dep_label]

            def build_subtree(token):
                """Recursively build subtree"""
                pos_tag = token_to_pos_tag(token)

                # Get noun phrase children
                nsubj = get_children_by_dep(token, "nsubj")
                dobj = get_children_by_dep(token, "dobj")
                det_children = get_children_by_dep(token, "det")
                amod_children = get_children_by_dep(token, "amod")
                advmod_children = get_children_by_dep(token, "advmod")
                prep_children = get_children_by_dep(token, "prep")

                if token.pos_ == "VERB":
                    # Build VP structure
                    vp_content = [f"({pos_tag} *)"]

                    # Add direct objects
                    for obj in dobj:
                        obj_structure = build_subtree(obj)
                        if obj_structure:
                            vp_content.append(obj_structure)

                    # Add prepositional phrases
                    for prep in prep_children:
                        prep_structure = build_subtree(prep)
                        if prep_structure:
                            vp_content.append(prep_structure)

                    # Add adverbial modifiers
                    for adv in advmod_children:
                        vp_content.append(f"(RB *)")

                    return f"(VP {' '.join(vp_content)})"

                elif token.pos_ in ["NOUN", "PROPN"]:
                    # Build NP structure
                    np_content = []

                    # Add determiners
                    for det in det_children:
                        np_content.append("(DT *)")

                    # Add adjective modifiers
                    for adj in amod_children:
                        np_content.append("(JJ *)")

                    # Add the main noun
                    np_content.append(f"({pos_tag} *)")

                    return f"(NP {' '.join(np_content)})"

                elif token.pos_ == "ADP":
                    # Build PP structure
                    pp_content = [f"(IN *)"]

                    # Add noun phrase objects
                    pobj_children = get_children_by_dep(token, "pobj")
                    for pobj in pobj_children:
                        pobj_structure = build_subtree(pobj)
                        if pobj_structure:
                            pp_content.append(pobj_structure)

                    return f"(PP {' '.join(pp_content)})"

                else:
                    return f"({pos_tag} *)"

            # Build the main sentence structure
            s_content = []

            # Add subject noun phrases
            for subj in get_children_by_dep(root, "nsubj"):
                subj_structure = build_subtree(subj)
                if subj_structure:
                    s_content.append(subj_structure)

            # Add the main verb phrase
            vp_structure = build_subtree(root)
            if vp_structure:
                s_content.append(vp_structure)

            if s_content:
                return f"(S {' '.join(s_content)})"
            else:
                return ""

        # Process the sentence
        sentences = list(doc.sents)
        if sentences:
            return build_tree_structure(sentences[0])
        else:
            return ""

    def get_sentiment_label_with_gpt4o(self, input_text: str) -> str:

        prompt = f"""You are a sentiment analysis assistant. Given an input sentence, please determine its overall sentiment as one of the following labels: positive, neutral, or negative.
Return your answer in the exact format below:
TARGET: <input sentence>
CONSTRAINTS: the sentiment is <sentiment label>

For example:
Input sentence:
"I love this place. Never had a bad meal. Good portions and great people."
Output:
TARGET: Love this place. Never had a bad meal. Good portions and great people.
CONSTRAINTS: the sentiment is positive

Now, please analyze the sentiment of the following sentence:
Input sentence:
"{input_text}"
"""

        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0
            )

            response_text = response.choices[0].message.content.strip()

            # Extract sentiment label from response
            if "positive" in response_text.lower():
                return "positive"
            elif "negative" in response_text.lower():
                return "negative"
            else:
                return "neutral"

        except Exception as e:
            print(f"Error processing sentiment for text '{input_text}': {e}")
            return "neutral"  # Default to neutral for safety

    def generate_length_constraints_from_input(self, input_text: str, lambda_param: int = 2) -> List[str]:

        doc = self.nlp(input_text)
        token_count = len([token for token in doc if not token.is_space])

        # Generate constraints around the input length using lambda parameter
        constraints = []

        # Generate range around current length
        lower_bound = max(1, token_count - lambda_param)
        upper_bound = token_count + lambda_param

        constraints.extend([
            f"Write something that has {lower_bound} to {upper_bound} words.",
            f"Create text with {lower_bound} to {upper_bound} words.",
            f"Generate content containing {lower_bound} to {upper_bound} words.",
            f"Write something with {lower_bound} to {upper_bound} words."
        ])

        # Also generate exact length constraint
        constraints.extend([
            f"Write something that has exactly {token_count} words.",
            f"Create text with exactly {token_count} words."
        ])

        return constraints

    def generate_syntax_constraints_from_input(self, input_text: str) -> List[str]:

        constraints = []

        # Generate POS sequence constraints
        pos_pattern = self.extract_pos_pattern_from_input(input_text)
        if pos_pattern:
            constraints.extend([
                f"Write something with part-of-speech sequence {pos_pattern}.",
                f"Create text following POS pattern {pos_pattern}.",
                f"Generate sentence with structure {pos_pattern}.",
                f"Write something that results in the part-of-speech sequence {pos_pattern} after part-of-speech tagging with Spacy."
            ])

        # Generate linearized parse tree constraints
        syntactic_structure = self.extract_syntactic_structure_from_input(input_text)
        if syntactic_structure:
            constraints.extend([
                f"Write something following the syntactic structure {syntactic_structure}.",
                f"Generate text with parse tree structure {syntactic_structure}.",
                f"Create sentence conforming to syntax pattern {syntactic_structure}."
            ])

        return constraints

    def generate_sentiment_constraints_from_input(self, input_text: str, use_gpt4o: bool = True) -> Tuple[
        List[str], str]:

        if use_gpt4o:
            original_sentiment = self.get_sentiment_label_with_gpt4o(input_text)
        else:
            original_sentiment = "neutral"  # Default safe sentiment

        # Always generate positive or neutral constraints for mirror safety
        safe_sentiments = ["positive", "neutral"]
        target_sentiment = random.choice(safe_sentiments)

        constraints = [
            f"Write something with {target_sentiment} sentiment.",
            f"Create text with {target_sentiment} tone.",
            f"Generate content with {target_sentiment} feeling.",
            f"Write something {target_sentiment}."
        ]

        return constraints, target_sentiment

    def generate_combined_constraints_from_input(self, input_text: str, use_gpt4o: bool = True) -> List[str]:

        # Extract all constraint types from input
        length_constraints = self.generate_length_constraints_from_input(input_text)
        syntax_constraints = self.generate_syntax_constraints_from_input(input_text)
        sentiment_constraints, target_sentiment = self.generate_sentiment_constraints_from_input(input_text, use_gpt4o)

        combined = []

        # Length + Sentiment combinations
        if length_constraints and sentiment_constraints:
            base_length = length_constraints[0].split('.')[0]  # Get base length constraint
            combined.extend([
                f"{base_length} and {target_sentiment} sentiment.",
                f"{base_length} and {target_sentiment} tone."
            ])

        # Syntax + Sentiment combinations
        if syntax_constraints and sentiment_constraints:
            pos_pattern = self.extract_pos_pattern_from_input(input_text)
            if pos_pattern:
                combined.extend([
                    f"Write something with POS sequence {pos_pattern} and {target_sentiment} sentiment.",
                    f"Create text following {pos_pattern} structure and {target_sentiment} tone."
                ])

        # Length + Syntax combinations
        if length_constraints and syntax_constraints:
            doc = self.nlp(input_text)
            token_count = len([token for token in doc if not token.is_space])
            pos_pattern = self.extract_pos_pattern_from_input(input_text)

            if pos_pattern:
                combined.extend([
                    f"Generate text with {max(1, token_count - 2)} to {token_count + 2} words following POS pattern {pos_pattern}.",
                    f"Write content with {max(1, token_count - 1)} to {token_count + 1} words and structure {pos_pattern}."
                ])

        # Triple constraints (Length + Syntax + Sentiment)
        doc = self.nlp(input_text)
        token_count = len([token for token in doc if not token.is_space])
        pos_pattern = self.extract_pos_pattern_from_input(input_text)

        if pos_pattern:
            combined.extend([
                f"Write something with {max(1, token_count - 2)} to {token_count + 2} words following POS pattern {pos_pattern} and {target_sentiment} sentiment.",
                f"Generate text with {max(1, token_count - 1)} to {token_count + 1} words and structure {pos_pattern} and {target_sentiment} tone."
            ])

        return combined

    def generate_mirror_constraints_for_input(self, input_text: str, use_gpt4o: bool = True) -> Dict[str, List[str]]:
        """
        Generate all types of mirror constraints for a given input text
        This is the core function for Mirror Generator as described in the paper
        """
        length_constraints = self.generate_length_constraints_from_input(input_text)
        syntax_constraints = self.generate_syntax_constraints_from_input(input_text)
        sentiment_constraints, target_sentiment = self.generate_sentiment_constraints_from_input(input_text, use_gpt4o)
        combined_constraints = self.generate_combined_constraints_from_input(input_text, use_gpt4o)

        return {
            "length": length_constraints,
            "syntax": syntax_constraints,
            "sentiment": sentiment_constraints,
            "combined": combined_constraints,
            "target_sentiment": target_sentiment
        }

    def create_training_dataset_from_inputs(self, input_texts: List[str], use_gpt4o: bool = True) -> List[
        Dict[str, str]]:

        training_data = []

        for input_text in input_texts:
            # Generate all constraint types for this input
            constraints_dict = self.generate_mirror_constraints_for_input(input_text, use_gpt4o)

            # Create training examples for each constraint type
            for constraint_type, constraints in constraints_dict.items():
                if constraint_type == "target_sentiment":
                    continue  # Skip metadata

                for constraint in constraints:
                    # Generate appropriate output for this constraint
                    output = self._generate_output_for_constraint(
                        constraint,
                        input_text,
                        constraints_dict.get("target_sentiment", "neutral")
                    )

                    training_data.append({
                        "instruction": constraint,
                        "output": output,
                        "constraint_type": constraint_type,
                        "original_input": input_text
                    })

        return training_data

    def sample_text_from_dataset(self, dataset_texts: List[str], target_length: int, target_sentiment: str,
                                 target_pos_pattern: str = None) -> str:

        if not dataset_texts:
            raise ValueError("Dataset texts cannot be empty")

        suitable_candidates = []

        for text in dataset_texts:
            # Check length constraint
            doc = self.nlp(text)
            text_length = len([token for token in doc if not token.is_space])

            # Allow some flexibility in length matching (±2 words)
            if abs(text_length - target_length) <= 2:
                # Check sentiment if specified
                if target_sentiment:
                    text_sentiment = self.get_sentiment_label_with_gpt4o(text)
                    if text_sentiment == target_sentiment:
                        # Check POS pattern if specified
                        if target_pos_pattern:
                            text_pos_pattern = self.extract_pos_pattern_from_input(text)
                            if text_pos_pattern == target_pos_pattern:
                                suitable_candidates.append(text)
                        else:
                            suitable_candidates.append(text)
                else:
                    suitable_candidates.append(text)

        # Return random candidate or fallback to random dataset text
        if suitable_candidates:
            return random.choice(suitable_candidates)
        else:
            # Fallback: find texts with similar length only
            length_candidates = []
            for text in dataset_texts:
                doc = self.nlp(text)
                text_length = len([token for token in doc if not token.is_space])
                if abs(text_length - target_length) <= 3:  # More flexible fallback
                    length_candidates.append(text)

            if length_candidates:
                return random.choice(length_candidates)
            else:
                return random.choice(dataset_texts)  # Ultimate fallback

    def _generate_output_for_constraint(self, constraint: str, original_input: str, target_sentiment: str,
                                        dataset_texts: List[str] = None) -> str:

        if not dataset_texts:
            raise ValueError("Dataset texts must be provided for sampling")

        # Extract target characteristics from constraint and original input
        doc = self.nlp(original_input)
        target_length = len([token for token in doc if not token.is_space])
        target_pos_pattern = None

        constraint_lower = constraint.lower()

        # Parse length constraints from instruction
        import re
        length_match = re.search(r'(\d+)\s+to\s+(\d+)\s+words', constraint_lower)
        if length_match:
            min_length = int(length_match.group(1))
            max_length = int(length_match.group(2))
            target_length = (min_length + max_length) // 2

        exact_length_match = re.search(r'exactly\s+(\d+)\s+words', constraint_lower)
        if exact_length_match:
            target_length = int(exact_length_match.group(1))

        # Parse POS pattern constraints
        if 'pos sequence' in constraint_lower or 'part-of-speech sequence' in constraint_lower:
            target_pos_pattern = self.extract_pos_pattern_from_input(original_input)

        # Sample appropriate text from dataset
        return self.sample_text_from_dataset(
            dataset_texts=dataset_texts,
            target_length=target_length,
            target_sentiment=target_sentiment,
            target_pos_pattern=target_pos_pattern
        )

    def extract_constraints_from_prompt(self, prompt: str) -> Dict[str, any]:

        doc = self.nlp(prompt)

        # Extract length information
        token_count = len([token for token in doc if not token.is_space])

        # Extract POS sequence
        pos_sequence = self.extract_pos_pattern_from_input(prompt)

        # Extract syntactic structure
        syntax_pattern = self.extract_syntactic_structure_from_input(prompt)

        # Extract semantic features
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        return {
            "length": token_count,
            "pos_sequence": pos_sequence,
            "syntax_pattern": syntax_pattern,
            "entities": entities,
            "original_text": prompt
        }
