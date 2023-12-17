# views.py
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
import base64

from .models import Todo
from .serializers import TodoSerializer, PatataSerializer
from . import appCV

@api_view(["GET"])
def request_blood(request):
    data = request.GET.get('img')
    print(data)
    img = base64.b64decode(data)
    
    #print(data)

    result = appCV.computeBloodAmount(img)

    print(result)

    rett =  {'scanResult':result}

    return JsonResponse(rett, safe=False)


def todo_list(request): 
    queryset = Todo.objects.all()

    serializer_class = TodoSerializer(queryset, many=True)

    return JsonResponse(serializer_class.data, safe=False)
