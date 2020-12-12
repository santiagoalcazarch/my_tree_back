from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import status, permissions
from rest_framework import mixins, generics

from .forms import UploadFileForm
from .models import Histogram, ColorDescriptor, Tamura

import base64
import glob

class CBIRView(APIView):
    def post(self, request):
        form = UploadFileForm(request.POST, request.FILES)
        self.handle_uploaded_file(request.FILES['file'])
        print("Image has been saved!")
        # Histograma
        print("Histogram...")
        res1 = Histogram().run()
        print(res1)
        # Descriptor
        print("Color descriptor...")
        res2 = ColorDescriptor((8, 12, 3)).run()
        print(res2)
        # Tamura
        print("Tamura...")
        res3 = Tamura().run()
        print(res3)

        results = [res1[0], res2[0], res3[0]]
        best = max(results)
        kind = None

        if best == res1[0]:
            kind = res1[1]
        if best == res2[0]:
            kind = res2[1]
        if best == res3[0]:
            kind = res3[1]

        print("Getting images...")
        imagesPath = "./archive/leafsnap-dataset/dataset/images/field/" + kind + "/*.jpg"
        n = 4
        count = 0
        images = []
        for imagePath in glob.glob(imagesPath):
            if count == n: break
            with open(imagePath, "rb") as img_file:
                images.append(base64.b64encode(img_file.read()))
            
        return Response({
            'similitude': best * 100,
            'kind': kind,
            'images': images
        })

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
