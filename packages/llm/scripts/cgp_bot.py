from nqs.llm.rag import ModelName, get_cgp_bot_graph

app = get_cgp_bot_graph()

while True:
    print("type in your question:")
    user_question = input()
    input_state = {
        "question": user_question,
        "grade_model": ModelName.PALM_2_VERTEXAI,
        "generate_model": ModelName.PALM_2_VERTEXAI,
        "k_best_docs": 4,
        "min_graded_docs": 1,
        "max_docs_for_generation": 3,
    }
    output = app.invoke(input_state)

    output_gen = output.get("output", "No final generation produced.")
    print("Final Generation:")
    if isinstance(output_gen, list):
        questions_str = "\n- ".join(output_gen)
        print(f"generated example questions:\n{questions_str}")
    else:
        print(output_gen)

    print("press any key to continue")
    input()
