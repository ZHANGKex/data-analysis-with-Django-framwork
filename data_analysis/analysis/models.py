from django.db import models

# Create your models here.
class StockData(models.Model):
    date = models.DateField(null=True, blank=True)
    year = models.IntegerField(null=True, blank=True)
    open_price = models.FloatField(null=True, blank=True)
    high_price = models.FloatField(null=True, blank=True)
    low_price = models.FloatField(null=True, blank=True)
    close_price = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    adj_close = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Stock Data for Year {self.year} and date {self.date}"