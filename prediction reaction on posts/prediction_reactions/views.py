from prediction_reactions.controller.category_controller import get_category_controller
from rest_framework.decorators import api_view
from rest_framework.response import Response

from prediction_reactions.controller.reactions_controller import get_reactions_controller


@api_view(['GET'])
def get_category_request(request):
    return Response({"result": get_category_controller(request)})

@api_view(['GET'])
def get_reactions_request(request):
    return Response({"result": get_reactions_controller(request)})
