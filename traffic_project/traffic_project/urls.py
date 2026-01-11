from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views # <--- Add this import
from core.views import dashboard

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Login Route: Points to our new template
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    
    # Logout Route: Redirects to login page after logging out
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    
    path('', dashboard, name='dashboard'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)