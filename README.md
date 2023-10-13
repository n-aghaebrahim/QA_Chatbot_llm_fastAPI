# Chatbot with OpenAI GPT-3.5 Turbo

This repository contains code for a chatbot powered by OpenAI's GPT-3.5 Turbo. The chatbot can answer questions and engage in conversations with users. Users are required to provide their own OpenAI API credentials to use this chatbot.

## Setup

1. **Clone the Repository**

   Clone this repository to your local machine using the following command:
   git clone <repository_url>


2. Set up a Python environment and install the required packages using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt


3. **Provide OpenAI API Key**

In the `llm.py` file, replace the placeholder `"sk-xxxxxxxxxxxxx"` with your actual OpenAI API key. You can obtain the API key by signing up for OpenAI services.

## Running the Application

To start the FastAPI application, use the following command:

uvicorn main:app --reload



This will launch the chatbot interface at `http://localhost:8000`.

## Usage

- Type your questions or messages in the chat interface on the web page.

- Click the "Ask" button to submit your query to the chatbot.

- The chatbot will provide responses based on the input you provide.

- You can have conversations with the chatbot, and it will respond accordingly.

## Important Note

- Users must provide their own OpenAI API credentials to use this chatbot. Replace the placeholder API key with your actual key in the `llm.py` file.

## Credits

This chatbot is powered by OpenAI's GPT-3.5 Turbo model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Please feel free to reach out if you have any questions or need further assistance.


