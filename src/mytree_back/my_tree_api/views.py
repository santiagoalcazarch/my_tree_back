from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import status, permissions
from rest_framework import mixins, generics

from .forms import UploadFileForm
from .models import Histogram, ColorDescriptor

class CBIRView(APIView):
    def post(self, request):
        form = UploadFileForm(request.POST, request.FILES)
        self.handle_uploaded_file(request.FILES['file'])
        print("Image has been saved!")
        # Histograma
        print("Histogram...")
        Histogram().run()
        # Descriptor
        print("Color descriptor...")
        ColorDescriptor((8, 12, 3)).run()
        return Response({'message': 'success!'})

    def handle_uploaded_file(self, f):
        destination = open('image.jpg', 'wb+')
        for chunk in f.chunks():
            destination.write(chunk)
        destination.close()
        # image_serializer = ImageSeralizer(data=request.data)
        # print(request.data)
        # if image_serializer.is_valid():
        #     image_serializer.save()
        #     return Response(image_serializer.data, status=status.HTTP_201_CREATED)
        # else:
        #     return Response(image_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
