# models.py
from django.db import models


class Todo(models.Model):
    #name = models.CharField(max_length=16)
    scanResult = models.CharField(max_length=16)

    #def __str__(self):
    #    return self.name

class Patata(models.Model):
    img = models.CharField(max_length=16)