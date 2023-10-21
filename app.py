import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PyPDF2 import PdfReader

st.title('PDF Chat App')
st.markdown('LLM Based PDF Chat App')
st.write("Made ❤️ with Rosary Abilash")

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define a padding token
tokenizer.pad_token = tokenizer.eos_token

def main():
    pdf = st.file_uploader('Upload PDF file', type='pdf')
    if pdf:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into paragraphs (you can customize this as needed)
        paragraphs = text.split('\n\n')

        question = st.text_input('Ask a question:')
        if st.button('Ask') and question:
            answer = ask_question(question, paragraphs)
            st.write('Question:', question)
            st.write('Answer:', answer)  
            

def ask_question(question, paragraphs):
    # Concatenate paragraphs as a single string
    text = "\n\n".join(paragraphs)

    # Tokenize the question and text
    inputs = tokenizer(question, text, return_tensors="pt", max_length=1024, truncation=True, padding=True)

    # Generate the answer using the model
    output = model.generate(**inputs)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

if __name__ == '__main__':
    main()
