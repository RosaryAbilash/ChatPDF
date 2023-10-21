import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PyPDF2 import PdfReader

st.title('PDF Chat App')
st.markdown('LLM Based PDF Chat App')

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can specify other GPT-2 variants like "gpt2-medium" or "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define a padding token
tokenizer.pad_token = tokenizer.eos_token

def main():
    st.header('PDF Chat App')
    pdf = st.file_uploader('Upload PDF file', type='pdf')

    if pdf:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into paragraphs (you can customize this as needed)
        paragraphs = text.split('\n\n')

        questions = st.text_area('Ask a question (separate multiple questions with line breaks):')
        question_list = questions.split('\n')

        if st.button('Ask'):
            if question_list:
                for question in question_list:
                    answer = ask_question(question, paragraphs)
                    st.write('Question:', question)
                    st.write('Answer:', answer)  # Display the answer to the user

def ask_question(question, paragraphs):
    # Concatenate paragraphs as a single string
    text = "\n\n".join(paragraphs)

    # Tokenize the question and text
    inputs = tokenizer(question, text, return_tensors="pt", max_length=1024, truncation=True, padding=True)

    # Generate the answer using the model
    output = model.generate(**inputs, num_return_sequences=1)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

if __name__ == '__main__':
    main()
