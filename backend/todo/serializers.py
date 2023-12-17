# serializers.py
from rest_framework.serializers import ModelSerializer
from .models import Todo, Patata

class TodoSerializer(ModelSerializer):
    class Meta:
        model = Todo
        fields = ['id', 'scanResult']

class PatataSerializer(ModelSerializer):
    class Meta:
        model = Patata
        fields = ['id', 'img']