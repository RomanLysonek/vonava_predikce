Zadání
Vytvořte predikci prodané Quantity (dohromady za oba prodejní kanály web + app), pro 30 časových řad (každá má vlastní ProductId) na následujících 7 dnů (všechna potřebná data jsou v souboru test_data.parquet). Pro natrénování modelu použíte data ze souboru train_data.parquet.

Popis sloupců:
ProductId – Identifikátor produktu 
DateKey – Časový identifikátor dne v roce
ProductAvailable – True – Zákazník si v daný den mohl produkt koupit 
		           False – Zákazník si v daný den nemohl produkt koupit 
QuantityApp – Prodaná Quantita v Notino aplikaci
QuantityWeb – Prodaná Quantity na Notino webu
IsSaleOrPromo – Produkt je ve štítkové akci 
  






CampaignSubTypeWeb – Identikátor kuponové nabídky na webu 
 
-1 –  Žádná kampaň 
0 – Malá kampaň 
1 – Sleva na všechno	
2 – Sleva na vybranou podkategorii
3 – BrandSale
4 – Stupňovaná sleva 
5 – Sleva při nákupu xy kusů 
	16 – Sleva na vybrané produkty 
	18 – Black Friday
	19 – Summer Black Friday
DiscountValueWebRelative– Velikost kampaňové slevy na webu v % 
CampaignSubTypeApp – Identikátor kuponové nabídky v aplikaci
-1 –  Žádná kampaň 
0 – Malá kampaň 
1 – Sleva na všechno	
2 – Sleva na vybranou podkategorii
3 – BrandSale
4 – Stupňovaná sleva 
5 – Sleva při nákupu xy kusů 
	16 – Sleva na vybrané produkty 
	18 – Black Friday
	19 – Summer Black Friday

DiscountValueAppRelative– Velikost kampaňové slevy v appce v %
PriceLocalVat – Prodejní cena bez započtení Slevy z promo akce 

Uvědomujeme si, že standardním přístupem pro tento typ úlohy jsou stromové modely (XGBoost, LightGBM apod.). Rádi bychom ale viděli i jiný přístup. 
