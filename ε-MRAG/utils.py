import os
import pdfplumber

def pdf_to_images(pdf_path, zoomin=3, page_from=0, page_to=299):
    images = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[page_from:page_to]):
                image = page.to_image(resolution=72 * zoomin).original
                images.append(image)
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return images


def save_images(images, save_dir):
    for i, img in enumerate(images):
        save_dir== os.path.join(save_dir, "images")
        img.save(f"page_{i + 1}.png")