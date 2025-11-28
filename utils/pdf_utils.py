import pdfplumber
from io import BytesIO
import base64

def extract_pdf_text(buffer: bytes):
    with pdfplumber.open(BytesIO(buffer)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def generate_pdf(text_content):
    """
    Generate a simple PDF containing the given text and return as base64 data URI.
    
    Args:
        text_content: Text to include in the PDF
    
    Returns:
        base64 data URI string (data:application/pdf;base64,...)
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Add text at position (100, 750)
        c.drawString(100, 750, str(text_content))
        
        c.save()
        buffer.seek(0)
        
        pdf_bytes = buffer.getvalue()
        return "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode()
    
    except ImportError:
        # Fallback: create minimal PDF manually
        text = str(text_content)
        pdf_content = f"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>\nendobj\n4 0 obj\n<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>\nendobj\n5 0 obj\n<< /Length {len(text) + 50} >>\nstream\nBT\n/F1 12 Tf\n100 750 Td\n({text}) Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000214 00000 n\n0000000304 00000 n\ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{400 + len(text)}\n%%EOF"
        return "data:application/pdf;base64," + base64.b64encode(pdf_content.encode()).decode()
    
    except Exception as e:
        raise Exception(f"PDF generation error: {e}")
