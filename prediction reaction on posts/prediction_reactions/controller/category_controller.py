from prediction_reactions.service.feature_extraction.model_category.predict_category import get_category


def get_category_controller(request):
    data = request.data
    sentences = data['sentences']
    result = get_category(sentences)
    return result
