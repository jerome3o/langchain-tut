from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI


def main():
    llm = OpenAI(temperature=0)
    tools = load_tools(['serpapi', 'llm-math'], llm=llm)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,
    )

    output = agent(
        "What date was langchain released?"
    )
    print(output)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
