from pprint import pprint

from src.agents_src.crew import crew


# input_data = {
#     "user_query": "Explain about adaptive radiation",
#     "chat_history": "{}"
# }
 
input_data = {
    "user_query": "Explain about Evolution and Ecosystem",
    "chat_history": {}
}

result = crew.kickoff(input_data)

result_dict = result.to_dict()

pprint(result_dict)