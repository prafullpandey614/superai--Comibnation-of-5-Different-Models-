from django.urls import path
from .views import *

urlpatterns = [
    path('', Home.as_view()),
    path('interact/',PromptInteractionAPIView.as_view(),name='chatbot_api')
]
