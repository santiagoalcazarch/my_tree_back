from django.urls import path

from . import views

urlpatterns = [
    path('cbir/', views.CBIRView.as_view())
]