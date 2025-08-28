"""
Example: Using Base64 Images in Lamoom Prompts

This example demonstrates how to add base64-encoded images to prompts
for use with vision models like GPT-4V.
"""

from lamoom import Prompt, Lamoom
import os

# Initialize the Lamoom client
client = Lamoom(openai_key=os.getenv("OPENAI_API_KEY"))

# Example 1: Single base64 image
def single_image_example():
    """Example of adding a single base64 image to a prompt."""
    
    # Create a prompt for image analysis
    image_prompt = Prompt(id="image_analysis")
    
    # Mock base64 image data (in practice, you'd load an actual image)
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    # Add text instruction
    image_prompt.add("Analyze this image and describe what you see:", role="user")
    
    # Add the base64 image
    image_prompt.add(base64_image, type='base64_image')
    
    print("Single image prompt created successfully!")
    print("To use this prompt with a vision model:")
    print("response = client.call(image_prompt.id, {}, 'openai/gpt-4-vision-preview')")
    
    return image_prompt

# Example 2: Multiple base64 images
def multiple_images_example():
    """Example of adding multiple base64 images to a prompt."""
    
    # Create a prompt for comparing multiple images
    multi_image_prompt = Prompt(id="multi_image_analysis")
    
    # Add text instruction
    multi_image_prompt.add("Compare these screenshots and identify the differences:", role="user")
    
    # Add placeholder for multiple images
    multi_image_prompt.add("screens", type='base64_image', is_multiple=True)
    
    print("Multi-image prompt created successfully!")
    print("To use this prompt with multiple images:")
    print("context = {'screens': [base64_image1, base64_image2, base64_image3]}")
    print("response = client.call(multi_image_prompt.id, context, 'openai/gpt-4-vision-preview')")
    
    return multi_image_prompt

# Example 3: Mixed content (text + images)
def mixed_content_example():
    """Example of mixing text and image content."""
    
    # Create a prompt with mixed content
    mixed_prompt = Prompt(id="mixed_content_analysis")
    
    # Add text instruction
    mixed_prompt.add("Please analyze the following:", role="user")
    
    # Add some text content
    mixed_prompt.add("1. Text description: This is a screenshot of a web application.", role="user")
    
    # Add base64 image
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    mixed_prompt.add(base64_image, type='base64_image')
    
    # Add more text
    mixed_prompt.add("2. Please identify any UI issues or improvements needed.", role="user")
    
    print("Mixed content prompt created successfully!")
    print("This prompt combines text and image content for comprehensive analysis.")
    
    return mixed_prompt

if __name__ == "__main__":
    print("=== Lamoom Base64 Image Examples ===\n")
    
    # Run examples
    single_image_example()
    print()
    
    multiple_images_example()
    print()
    
    mixed_content_example()
    print()
    
    print("=== Key Points ===")
    print("- Use type='base64_image' to specify image content")
    print("- Use is_multiple=True for multiple images via context")
    print("- Images are formatted as data URLs for vision models")
    print("- Token calculation accounts for image content appropriately") 