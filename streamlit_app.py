import streamlit as st

from src.generator import generate_email
from src.models import get_client


def main() -> None:
    st.set_page_config(
        page_title="Email Generation Assistant",
        page_icon="📧",
        layout="centered",
    )

    st.title("Email Generation Assistant")
    st.markdown(
        "Use this interface to generate a professional business email from user-provided "
        "Intent, Key Facts, and Tone. The app uses OpenAI GPT-4o for generation."
    )

    intent = st.text_input(
        "Email Intent",
        placeholder="e.g. Follow up after meeting and propose next steps",
        help="Describe the core purpose of the email.",
    )

    tone = st.text_input(
        "Tone",
        value="Warm and professional",
        help="Describe the desired tone (formal, casual, urgent, empathetic, etc.).",
    )

    key_facts_input = st.text_area(
        "Key Facts (one per line)",
        placeholder="Enter each fact on its own line, for example:\n- Meeting held on June 12th at Nexus HQ\n- Client interested in cloud storage integration\n- Propose a 30-day pilot starting July 1st",
        help="Provide the facts that must appear in the generated email.",
        height=180,
    )

    if st.button("Generate Email"):
        if not intent.strip() or not tone.strip() or not key_facts_input.strip():
            st.warning("Please provide the email intent, key facts, and tone before generating.")
            return

        key_facts = [line.strip() for line in key_facts_input.splitlines() if line.strip()]

        try:
            client, config = get_client("gpt-4o")
            generated_email = generate_email(client, config, intent, key_facts, tone)

            st.subheader("Generated Email")
            st.text_area("", value=generated_email, height=320)

            st.markdown("---")
            st.markdown(
                "#### Notes"
                "\n- The email is generated using OpenAI GPT-4o."
                "\n- Ensure your `OPENAI_API_KEY` is configured in `.env`."
                "\n- If the model response is slow, wait a few moments for the API call to complete."
            )

        except Exception as error:
            st.error(f"Failed to generate email: {error}")

    st.sidebar.header("Streamlit Email Generator")
    st.sidebar.write(
        "Enter an intent, tone, and key facts. Click Generate to create a professional email using GPT-4o."
    )
    st.sidebar.write("Ensure your `.env` contains a valid `OPENAI_API_KEY`.")


if __name__ == "__main__":
    main()
