from django.urls import path
from .views import get_predictions

urlpatterns = [
    path('predict/', get_predictions),
]
