from django.urls import path
from .views import get_predictions
from .views import predict_three_to_sixteen

urlpatterns = [
    path('predict/', get_predictions),
    path('reverse/',predict_three_to_sixteen)
]
