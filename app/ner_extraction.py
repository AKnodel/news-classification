# app/ner_extraction.py
import spacy
from itertools import combinations
import stanza

# Load Stanza and spaCy models
nlp_stanza = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,ner,depparse,coref')
nlp_spacy = spacy.load("en_core_web_lg")

# Coreference Resolution using Stanza
def get_coref_resolutions(text):
    doc = nlp_stanza(text)
    coref_chains = doc.coref
    return doc, coref_chains

def resolve_coreferences(text):
    doc, coref_chains = get_coref_resolutions(text)
    resolved_text = text
    
    # Dictionary to replace mentions with their representative text
    mention_to_replacement = {}
    
    for chain in coref_chains:
        representative_text = chain.representative_text
        for mention in chain.mentions:
            sentence_idx = mention.sentence
            sentence = doc.sentences[sentence_idx]
            start_word = mention.start_word
            end_word = mention.end_word
            
            # Join words from start_word to end_word
            try:
                mention_text = ' '.join(sentence.text.split()[start_word:end_word])
                mention_to_replacement[mention_text] = representative_text
            except IndexError:
                continue
    
    # Replace mentions with their representative text
    for mention, replacement in mention_to_replacement.items():
        resolved_text = resolved_text.replace(mention, replacement)
    return resolved_text

# Extract Character Names
def list_all_characters(text):
    doc = nlp_spacy(text)
    characters = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            characters.add(ent.text)
    return list(characters)

# Identify the Protagonist
def find_protagonist_with_context(text):
    doc = nlp_spacy(text)
    
    character_mentions = {}
    character_subject_counts = {}

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            character_name = ent.text
            if character_name not in character_mentions:
                character_mentions[character_name] = 0
                character_subject_counts[character_name] = 0

            character_mentions[character_name] += 1

            for token in doc:
                if token.text == character_name and token.dep_ == "nsubj":
                    character_subject_counts[character_name] += 1

    if character_mentions:
        protagonist = max(character_mentions, key=lambda name: (character_subject_counts[name], character_mentions[name]))
        return protagonist
    else:
        return None

# Extract Character Traits
def extract_character_traits(text, characters):
    # Process the text
    doc = nlp_spacy(text)
    
    character_traits = {char: [] for char in characters}

    # Loop through each sentence in the document
    for sent in doc.sents:
        sentence_characters = set()
        sentence_pairs = []

        # First, find all characters in the sentence
        for token in sent:
            if token.text in characters:
                sentence_characters.add(token.text)

        # Now, find noun-adjective pairs in the sentence
        for token in sent:
            if token.pos_ == "NOUN":
                for child in token.children:
                    if child.pos_ == "ADJ":
                        sentence_pairs.append(f"{child.text} {token.text}")
                        # Handle conjunctive adjectives
                        for conj in child.conjuncts:
                            if conj.pos_ == "ADJ":
                                sentence_pairs.append(f"{conj.text} {token.text}")

        # Associate each pair with the characters found in the sentence
        for char in sentence_characters:
            character_traits[char].extend(sentence_pairs)
    
    return character_traits

# Calculate Similarity between Traits
def calculate_similarity(unique_traits_1, unique_traits_2):
    # Simple similarity score based on the overlap between two sets
    if len(unique_traits_1) + len(unique_traits_2) == 0:
        return 0
    overlap = len(unique_traits_1.intersection(unique_traits_2))
    total = len(unique_traits_1.union(unique_traits_2))
    return overlap / total

# Determine Relationship based on Shared and Unique Traits
def determine_relationship(shared_traits, unique_traits_1, unique_traits_2):
    relationship_strength = len(shared_traits)
    similarity_score = calculate_similarity(unique_traits_1, unique_traits_2)
    
    # Adjusted relationship logic
    if relationship_strength > 1 or similarity_score > 0.3:
        return 'A strong, close relationship with deep mutual understanding and support.'
    elif relationship_strength > 2 or similarity_score > 0.2:
        return 'A neutral relationship with some shared traits but not much depth.'
    else:
        return 'A distant or strained relationship with little to no connection.'

# Analyze Relationships between Characters
def analyze_character_relationships(character_traits):
    relationships = {}
    character_pairs = combinations(character_traits.keys(), 2)
    for character_1, character_2 in character_pairs:
        traits_1 = set(character_traits[character_1])
        traits_2 = set(character_traits[character_2])
        
        shared_traits = traits_1.intersection(traits_2)
        unique_traits_1 = traits_1 - traits_2
        unique_traits_2 = traits_2 - traits_1
        
        relationship = determine_relationship(shared_traits, unique_traits_1, unique_traits_2)
        relationships[(character_1, character_2)] = relationship
    return relationships

# Main Function to Process Story
def analyze_story(text):
    # resolved_story = resolve_coreferences(text)
    # print("resolved story: ", resolved_story)
    characters = list_all_characters(text)
    protagonist = find_protagonist_with_context(text)
    character_traits = extract_character_traits(text, characters)
    relationships = analyze_character_relationships(character_traits)

    # Returning structured data
    return {
        "characters": characters,
        "protagonist": protagonist,
        "character_traits": character_traits,
        "relationships": relationships
    }

# Example usage:
if __name__ == "__main__":
    text = """
    In the village of Narnia, Emma, a meticulous and kind-hearted librarian, enjoyed her afternoons organizing the local book collection. 
    Her close friend, Leo, a lively and inventive artist, often joined her, bringing his imaginative flair to the library’s events. 
    Together, Emma and Leo had insightful conversations and transformed the library into a vibrant hub of activity, their collaboration marked by a seamless blend of creativity. 
    Their shared efforts made every event a success, from book readings to art displays. Meanwhile, Henry, a solitary and introspective gardener, tended to his plants nearby. 
    He admired their work from afar but rarely engaged, a contrast to Emma and Leo’s energetic partnership.
    """
    
    results = analyze_story(text)
    print(f"Characters: {results['characters']}")
    print(f"Protagonist: {results['protagonist']}")
    print("Character Traits:")
    for char, traits in results['character_traits'].items():
        print(f"{char}: {', '.join(traits)}")
    print("Character Relationships:")
    for (char1, char2), relationship in results['relationships'].items():
        print(f"{char1} and {char2}: {relationship}")
