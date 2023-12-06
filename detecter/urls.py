from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('upload/result/', views.result, name='result'),
    path('upload/result2/', views.result2, name='result2'),
    path('home/notice', views.notice, name='notice')

]