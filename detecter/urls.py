from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('upload/result/', views.result, name='result'),
    path('upload/result2/', views.result2, name='result2'),
    path('home/notice', views.notice, name='notice'),
    path('home/notice_view', views.notice_view, name='notice_view')

]

