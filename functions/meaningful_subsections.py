import json
import re
from typing import List, Dict, Any

def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization that approximates CLIP tokenization.
    Splits on whitespace and punctuation.
    """
    # Remove extra whitespace and split on spaces
    tokens = text.strip().split()
    
    # Further split on punctuation to better approximate CLIP tokenization
    refined_tokens = []
    for token in tokens:
        # Split on common punctuation while keeping the punctuation
        parts = re.split(r'([.,;:!?()])', token)
        for part in parts:
            if part.strip():
                refined_tokens.append(part.strip())
    
    return refined_tokens

def find_sentence_boundaries(text: str) -> List[int]:
    """
    Find sentence boundaries in the text.
    Returns list of indices where sentences end.
    """
    sentences = re.split(r'[.!?]+', text)
    boundaries = []
    current_pos = 0
    
    for sentence in sentences[:-1]:  # Exclude the last empty split
        current_pos += len(sentence) + 1  # +1 for the punctuation
        boundaries.append(current_pos)
    
    return boundaries

def divide_into_meaningful_subsections(text: str, max_tokens: int = 77) -> List[str]:
    """
    Divide text into meaningful subsections with maximum token count.
    
    Args:
        text: The input text to divide
        max_tokens: Maximum number of tokens per subsection (default: 77)
    
    Returns:
        List of text subsections, each with <= max_tokens
    """
    # Tokenize the entire text
    tokens = tokenize_text(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    # Find sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    subsections = []
    current_subsection = ""
    current_token_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_tokens = tokenize_text(sentence)
        sentence_token_count = len(sentence_tokens)
        
        # If adding this sentence would exceed max_tokens
        if current_token_count + sentence_token_count > max_tokens:
            # If we have content in current subsection, save it
            if current_subsection:
                subsections.append(current_subsection.strip())
                current_subsection = ""
                current_token_count = 0
            
            # If single sentence is longer than max_tokens, split it further
            if sentence_token_count > max_tokens:
                # Split long sentence into chunks
                sentence_chunks = split_long_sentence(sentence, max_tokens)
                subsections.extend(sentence_chunks)
            else:
                current_subsection = sentence
                current_token_count = sentence_token_count
        else:
            # Add sentence to current subsection
            if current_subsection:
                current_subsection += " " + sentence
            else:
                current_subsection = sentence
            current_token_count += sentence_token_count
    
    # Add remaining content
    if current_subsection:
        subsections.append(current_subsection.strip())
    
    return subsections

def split_long_sentence(sentence: str, max_tokens: int) -> List[str]:
    """
    Split a sentence that's longer than max_tokens into meaningful chunks.
    """
    tokens = tokenize_text(sentence)
    chunks = []
    
    # Split on natural boundaries like commas, semicolons
    parts = re.split(r'([,;])', sentence)
    current_chunk = ""
    current_token_count = 0
    
    for part in parts:
        if not part.strip():
            continue
            
        part_tokens = tokenize_text(part)
        part_token_count = len(part_tokens)
        
        if current_token_count + part_token_count <= max_tokens:
            current_chunk += part
            current_token_count += part_token_count
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = part
            current_token_count = part_token_count
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If still too long, split by word count
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = tokenize_text(chunk)
        if len(chunk_tokens) <= max_tokens:
            final_chunks.append(chunk)
        else:
            # Split by word boundaries
            words = chunk.split()
            current_words = []
            
            for word in words:
                word_tokens = tokenize_text(word)
                if len(current_words) + len(word_tokens) <= max_tokens:
                    current_words.append(word)
                else:
                    if current_words:
                        final_chunks.append(" ".join(current_words))
                    current_words = [word]
            
            if current_words:
                final_chunks.append(" ".join(current_words))
    
    return final_chunks

def process_example_entry():
    """
    Process the specific example entry from the dataset.
    """
    example_entry = {
        "3526431764_056d2c61dc.jpg": {
            "short_caption": "A woman and a man are walking through the woods, with the woman showing something to the man.",
            "long_detailed": "The image depicts an outdoor scene where two people are looking at their camera. They both seem to be engaged in photography activities on the dirt near some trees with bright sunshine illuminating them. It appears they could potentially be taking pictures or recording video footage. There is also another man standing nearby wearing sunglasses.\n\nSeveral personal items can also be observed within the scene, such as backpacks placed further away from the main subjects; one close to each side of the tree-lined area while there is another towards the right edge of the picture. Additionally, three handbags are scattered across the setting – one positioned near the bottom left corner, one located closer to the center, and another toward the top-right part of the frame."
        }
    }
    
    image_id = "3526431764_056d2c61dc.jpg"
    long_description = example_entry[image_id]["long_detailed"]
    
    print(f"Original text length: {len(tokenize_text(long_description))} tokens")
    print(f"Original text:\n{long_description}\n")
    print("-" * 80)
    
    # Divide into subsections
    subsections = divide_into_meaningful_subsections(long_description, max_tokens=77)
    
    print(f"Divided into {len(subsections)} subsections:")
    print("=" * 80)
    
    for i, subsection in enumerate(subsections, 1):
        token_count = len(tokenize_text(subsection))
        print(f"Subsection {i} ({token_count} tokens):")
        print(f"{subsection}\n")
        print("-" * 40)
    
    # Create the modified entry
    modified_entry = {
        image_id: {
            "short_caption": example_entry[image_id]["short_caption"],
            "long_detailed": long_description,  # Keep original
            "subsections": subsections,
            "num_subsections": len(subsections)
        }
    }
    
    return modified_entry

def process_dataset_entry(entry_data: Dict[str, Any], max_tokens: int = 77) -> Dict[str, Any]:
    """
    Process a single dataset entry to create meaningful subsections.
    
    Args:
        entry_data: Dictionary containing image data with captions
        max_tokens: Maximum tokens per subsection
    
    Returns:
        Modified entry with subsections added
    """
    modified_entry = {}
    
    for image_id, data in entry_data.items():
        long_description = data.get("long_detailed", "")
        
        # Check if we need to divide the text
        tokens = tokenize_text(long_description)
        
        if len(tokens) <= max_tokens:
            # No need to divide
            modified_entry[image_id] = {
                **data,
                "subsections": [long_description],
                "num_subsections": 1
            }
        else:
            # Divide into subsections
            subsections = divide_into_meaningful_subsections(long_description, max_tokens)
            modified_entry[image_id] = {
                **data,
                "subsections": subsections,
                "num_subsections": len(subsections)
            }
    
    return modified_entry

def analyze_token_distribution(dataset_path: str):
    """
    Analyze the token distribution in the dataset to understand how many entries need division.
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    token_counts = []
    long_entries = []
    
    for image_id, entry in data.items():
        long_description = entry.get("long_detailed", "")
        token_count = len(tokenize_text(long_description))
        token_counts.append(token_count)
        
        if token_count > 77:
            long_entries.append((image_id, token_count))
    
    print(f"Dataset Analysis:")
    print(f"Total entries: {len(data)}")
    print(f"Entries with > 77 tokens: {len(long_entries)}")
    print(f"Average token count: {sum(token_counts) / len(token_counts):.2f}")
    print(f"Max token count: {max(token_counts)}")
    print(f"Min token count: {min(token_counts)}")
    
    if long_entries:
        print(f"\nTop 10 longest entries:")
        sorted_long = sorted(long_entries, key=lambda x: x[1], reverse=True)[:10]
        for image_id, count in sorted_long:
            print(f"  {image_id}: {count} tokens")

def process_entire_dataset(input_path: str, output_path: str = None, max_tokens: int = 77):
    """
    Process the entire dataset to create meaningful subsections for all entries.
    
    Args:
        input_path: Path to the original all_captions.json file
        output_path: Path to save the modified dataset (optional)
        max_tokens: Maximum tokens per subsection
    
    Returns:
        Dictionary containing the processed dataset
    """
    if output_path is None:
        # Create output path in the same directory
        import os
        directory = os.path.dirname(input_path)
        output_path = os.path.join(directory, "mod_77_tokenized_all_captions.json")
    
    print(f"Loading dataset from: {input_path}")
    with open(input_path, 'r') as f:
        original_data = json.load(f)
    
    print(f"Processing {len(original_data)} entries...")
    
    processed_data = {}
    entries_requiring_division = 0
    total_subsections_created = 0
    
    for i, (image_id, entry_data) in enumerate(original_data.items()):
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(original_data)} entries...")
        
        long_description = entry_data.get("long_detailed", "")
        
        # Check if we need to divide the text
        tokens = tokenize_text(long_description)
        
        if len(tokens) <= max_tokens:
            # No need to divide - keep as single subsection
            processed_data[image_id] = {
                **entry_data,
                "subsections": [long_description],
                "num_subsections": 1,
                "original_token_count": len(tokens),
                "requires_division": False
            }
        else:
            # Only divide if tokens > 77
            subsections = divide_into_meaningful_subsections(long_description, max_tokens)
            processed_data[image_id] = {
                **entry_data,
                "subsections": subsections,
                "num_subsections": len(subsections),
                "original_token_count": len(tokens),
                "requires_division": True
            }
            entries_requiring_division += 1
            total_subsections_created += len(subsections)
    
    # Add metadata
    processed_data["_metadata"] = {
        "original_entries": len(original_data),
        "entries_requiring_division": entries_requiring_division,
        "total_subsections_created": total_subsections_created,
        "max_tokens_per_subsection": max_tokens,
        "processing_timestamp": time.time(),
        "average_subsections_per_divided_entry": total_subsections_created / max(entries_requiring_division, 1)
    }
    
    # Save the processed dataset
    print(f"Saving processed dataset to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("DATASET PROCESSING SUMMARY")
    print("="*80)
    print(f"Total entries processed: {len(original_data)}")
    print(f"Entries requiring division: {entries_requiring_division} ({entries_requiring_division/len(original_data)*100:.1f}%)")
    print(f"Entries with single subsection: {len(original_data) - entries_requiring_division}")
    print(f"Total subsections created: {total_subsections_created}")
    print(f"Average subsections per divided entry: {total_subsections_created/max(entries_requiring_division, 1):.2f}")
    print(f"Max tokens per subsection: {max_tokens}")
    print(f"Output saved to: {output_path}")
    
    return processed_data

def validate_processed_dataset(dataset_path: str, max_tokens: int = 77):
    """
    Validate the processed dataset to ensure all subsections are within token limits.
    
    Args:
        dataset_path: Path to the processed dataset
        max_tokens: Maximum allowed tokens per subsection
    
    Returns:
        Validation results dictionary
    """
    print(f"Validating dataset: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Extract metadata if present
    metadata = data.pop("_metadata", {})
    
    validation_results = {
        "valid_entries": 0,
        "invalid_entries": 0,
        "oversized_subsections": [],
        "empty_subsections": [],
        "token_distribution": []
    }
    
    for image_id, entry in data.items():
        if image_id == "_metadata":
            continue
            
        subsections = entry.get("subsections", [])
        entry_valid = True
        
        for i, subsection in enumerate(subsections):
            token_count = len(tokenize_text(subsection))
            validation_results["token_distribution"].append(token_count)
            
            if token_count > max_tokens:
                validation_results["oversized_subsections"].append({
                    "image_id": image_id,
                    "subsection_index": i,
                    "token_count": token_count,
                    "text": subsection[:100] + "..." if len(subsection) > 100 else subsection
                })
                entry_valid = False
            
            if token_count == 0:
                validation_results["empty_subsections"].append({
                    "image_id": image_id,
                    "subsection_index": i
                })
                entry_valid = False
        
        if entry_valid:
            validation_results["valid_entries"] += 1
        else:
            validation_results["invalid_entries"] += 1
    
    # Calculate statistics
    token_dist = validation_results["token_distribution"]
    validation_results["statistics"] = {
        "total_subsections": len(token_dist),
        "avg_tokens": sum(token_dist) / len(token_dist) if token_dist else 0,
        "max_tokens_found": max(token_dist) if token_dist else 0,
        "min_tokens_found": min(token_dist) if token_dist else 0,
        "tokens_within_limit": len([t for t in token_dist if t <= max_tokens]),
        "tokens_over_limit": len([t for t in token_dist if t > max_tokens])
    }
    
    # Print validation results
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Valid entries: {validation_results['valid_entries']}")
    print(f"Invalid entries: {validation_results['invalid_entries']}")
    print(f"Total subsections: {validation_results['statistics']['total_subsections']}")
    print(f"Average tokens per subsection: {validation_results['statistics']['avg_tokens']:.2f}")
    print(f"Max tokens found: {validation_results['statistics']['max_tokens_found']}")
    print(f"Subsections within {max_tokens} token limit: {validation_results['statistics']['tokens_within_limit']}")
    print(f"Subsections over limit: {validation_results['statistics']['tokens_over_limit']}")
    
    if validation_results["oversized_subsections"]:
        print(f"\nWarning: {len(validation_results['oversized_subsections'])} oversized subsections found!")
        for item in validation_results["oversized_subsections"][:5]:  # Show first 5
            print(f"  {item['image_id']}: {item['token_count']} tokens")
    
    if validation_results["empty_subsections"]:
        print(f"\nWarning: {len(validation_results['empty_subsections'])} empty subsections found!")
    
    return validation_results

if __name__ == "__main__":
    import time
    
    # Test with the example
    print("Processing example entry...")
    result = process_example_entry()
    
    # Save the result for inspection
    with open('/Users/damdam/Desktop/447 project/code/LightVision/data/example_subsections.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\nExample processing complete!")
    print("Result saved to: data/example_subsections.json")
    
    # Process the entire dataset
    print("\n" + "="*80)
    print("Processing entire dataset...")
    
    input_path = '/Users/damdam/Desktop/447 project/code/LightVision/data/all_captions.json'
    output_path = '/Users/damdam/Desktop/447 project/code/LightVision/data/mod_77_tokenized_all_captions.json'
    
    try:
        processed_dataset = process_entire_dataset(input_path, output_path, max_tokens=77)
        
        # Validate the processed dataset
        print("\n" + "="*80)
        print("Validating processed dataset...")
        validation_results = validate_processed_dataset(output_path, max_tokens=77)
        
        # Save validation results
        validation_output_path = '/Users/damdam/Desktop/447 project/code/LightVision/data/validation_results.json'
        with open(validation_output_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        print(f"\nValidation results saved to: {validation_output_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file: {input_path}")
        print("Please ensure the all_captions.json file exists in the data directory.")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
