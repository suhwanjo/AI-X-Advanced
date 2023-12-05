from django.shortcuts import render
from django.http import HttpResponse


def home(request):
    return render(request,'detecter/home.html')

def upload(request):
    return render(request,'detecter/upload.html')

def result(request):
    return render(request,'detecter/result.html')

def notice(request):
    return render(request,'detecter/notice.html')
