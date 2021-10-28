from prediction_reactions.service.feature_extraction.model_category.predict_category import get_vector_from_model_category


def get_reactions_controller(request):
    data = request.data
    sentences = data['sentences']
    result = get_vector_from_model_category(sentences)
    return result
