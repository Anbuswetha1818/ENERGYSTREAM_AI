import zipfile
import xml.etree.ElementTree as ET
import os

def extract_text(docx_path):
    if not os.path.exists(docx_path):
        print(f"Error: {docx_path} not found")
        return
    
    try:
        with zipfile.ZipFile(docx_path) as z:
            doc_xml = z.read('word/document.xml')
            root = ET.fromstring(doc_xml)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text_elements = root.findall('.//w:t', ns)
            text = ' '.join([el.text for el in text_elements if el.text])
            print(text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_text('ABSTRACT - ENERGY CONSUMPTION.docx')
