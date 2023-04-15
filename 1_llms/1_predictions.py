from langchain.llms import OpenAI

def main():
    llm = OpenAI(temperature=1)
    text = "What is a good name for my clownfish? I like open source software."

    print(llm(text))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
