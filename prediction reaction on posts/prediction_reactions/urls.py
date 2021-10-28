from django.urls import path

from prediction_reactions import views

urlpatterns = [
    # TeAccount URL
    path('get_category', views.get_category_request),
    path('get_reactions', views.get_reactions_request),

]
