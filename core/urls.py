
from django.contrib import admin
from django.urls import path,include
from django.conf.urls.static import static

from app.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index),
    path('allticket',allticket),
    path('predict',predictticket),
    path('finalresult/<str:ticker_value>/<str:number_of_days>/', finalresult),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)