from django.urls import path
from . import views

app_name = "capture"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/upload/", views.upload_image, name="upload_image"),
]
