from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def find_most_similar_strings(input_string, dataset):
    # Load the pre-trained BERT model
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Generate embeddings for the input string and the dataset strings
    input_embedding = model.encode([input_string])[0]
    dataset_embeddings = model.encode(dataset)

    # Compute cosine similarity between the input embedding and each dataset embedding
    similarities = cosine_similarity([input_embedding], dataset_embeddings)[0]

    # Combine similarities with dataset strings
    combined_results = list(zip(dataset, similarities))

    # Sort the combined results based on similarity scores in descending order
    combined_results.sort(key=lambda x: x[1], reverse=True)

    # Return the ten most similar strings and their similarity scores
    return combined_results[:10]


# Dataset of strings
dataset = [
    "As members of Leadership Initiatives (LI) International Internship Program, we have been working through our partnership with the Bauchi community, to address human rights abuses through advocacy and raising awareness through research, community workshops, and educational training programs. LI and the team aim to mobilize support, secure adequate funding, empower local organizations to combat human rights violations.",
    "Conflict continues to escalate in Israel and Palestine, and thousands of innocent children and civilians have been killed, injured, or taken hostage. Millions are facing a devastating humanitarian crisis in Gaza with a dire need for aid. Your donation to the Israel-Palestine Crisis Relief Fund will provide emergency relief and long-term support to people in need.",
    "By donating to this project, you contribute to the purchase of medical equipment, hospital furniture and the improvement of the physical infrastructure of the Children Hospital of San Vicente Foundation. Thus, every year, 60,000 children in Colombia, with serious illnesses, will continue to receive the best specialized medical care.",
    "Through this project we want to initiate a business model for poor rural women of 10 areas based on organic and commercial kitchen gardening to transform the low income women lives from poverty to prosperity. Women Vegetable Clubs (WVC) is an idea to club rural women for collective entrepreneurship through organic commercial kitchen gardening. As the project is highly sustainable therefore it will be initiated from from 1 rural area and will move towards the next 9 areas after the completion.",
    "Sudan is experiencing a catastrophic civil war as violent clashes between paramilitary and government forces threaten the lives of people across the country. Since April 2023, thousands of Sudanese have been killed, and millions more injured and displaced in the fighting. Your donation to the Sudan Emergency Fund will provide emergency relief, food, water, medicine, and other essential supplies to impacted communities.",
    "Two powerful earthquakes struck Turkey and Syria on Feb. 6, killing more than 50,000 people and injuring thousands more. Millions of survivors urgently need help. Your donation to the Turkey and Syria Earthquake Relief Fund will provide emergency relief and fuel long-term recovery efforts in Turkey and Syria.",
    "Devastating floods hit Libya eastern region on Monday, killing thousands of people, injuring thousands more, and causing widespread damage. Your donation to the Libya Flood Relief Fund will provide emergency relief and long-term support to affected communities.",
    "Millions of Haitians are confronting a complex crisis driven by political and economic turmoil, leading to widespread violence and food and water scarcity. Haiti is also struggling to recover from and prepare for hurricanes and earthquakes. Your donation to the Haiti Crisis Relief Fund will provide emergency assistance to people in need and support long-term response efforts led by local organizations who continue to deliver vital services.",
    "Isha Vidhya English-medium schools provide high quality school education to rural children in India who cannot otherwise access or afford it. The schools adopt a nurturing, holistic approach to education, helping children learn joyfully. 64% of the children get full sponsorship while rest pay a subsidized fee. Your donation goes towards critical infrastructure like learning material (including STEM), classrooms, school bus, etc. for 10 Isha Vidhya rural schools & Govt. Schools Support Program.",
    "Your child is visually impaired. What is the right thing to do? Where do you find help? Upon diagnosis of the visual impairment, parents are often in shock and needed time to come to terms with the situation. At St. Nicholas Home, parents of visually impaired children are encouraged to join the Early Intervention Programme (home-based) to help them to understand, garner support and receive counselling to help them c",
    "The food scarcity is a severe issue that impacts the optimal growth and development of young children that Albergue supports. With your help, we can make it possible for a child to eat three meals a day and the essential nutritional follow-up necessary for their integral development."
]

# Example usage:
input_string = "We are suffering from civil war please provide help."
most_similar_strings = find_most_similar_strings(input_string, dataset)
print("Ten Most Similar Strings:")
for string, similarity in most_similar_strings:
    print(f"String: {string}")
    print()


