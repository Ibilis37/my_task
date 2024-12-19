# Курсовая работа по Математическому моделированию
**Выполнила**: Комольцева Диана ВИтальевна ПМ-42
## Содержательная постановка задачи
### Задача о футболисте (с учетом радиуса мяча и учетом трения о воздух)
Постановка задачи:
1. Определить зависимость угла удара по вертикали от угла к створу ворот для голевой ситуации
2. Опредлить область на поле, откуда спортсмен попадает в створ ворот
3. Определить зависимость площади попадания от углов удара
4. Определить зависимость площади попадания от скорости и углов удара 

## Концептуальная постановка задачи
Необходимо составить математическую модель полета футбольного мяча с заданным радиусом и массой в условиях действия поля силы тяжести с учетом сопротивления воздуха. Необходимо исследовать, при каких параметрах (углах начальной скорости к горизонту и к вертикали, начальных координатах, величине начальной скорости) удар по мячу будет соответствовать удару при голевой ситуации.

![](./mm1.jpg)
### Основные параметры:
- $v_0$ — начальная скорость мяча
- $x_0, y_0, z_0$ — начальное положение мяча
- $\theta$ —  угол удара по горизонтали
- $\alpha$ — угол удара по вертикали
- $C_d$ — коэффициент сопротивления воздуха
- $\rho$ — плотность воздуха
- $A$ — площадь поперечного сечения мяча
- $m$ — масса мяча
- $r$ — радиус мяча
- $g$ — ускорение свободного падения
- Высота ворот $h = 2.44$ м
- Ширина ворот $w = 7,32м$
- Размеры поля $a = 100м, b = 64 м$

### Примем следующие допущения:
1. Рассмотрим мяч как материальную точку с массой m, радиусом r
2. Движение происходит в поле силы тяжести с постоянным ускорением свободного падения g=9,81 и описывается уравнениями классической механики Ньютона
3. На мяч действует сила сопростивления воздуха, вычисляемая по формуле:
$\vec{F_{сопр}} = -\frac{1}{2}C_dApV\vec{V}$
4. Мяч движется в воздушной среде с плотностью p = 1,2255 кг/м³. Коэффициент сопротивления формы $C_d$ примем постоянным и равным 0,47 (соответствует сфере). Площадь поперечного сечения мяча - площадь окружности с радисом r


![](0_2.png)

Для описания модели применим второй закон Ньютона в векторной форме:

$\vec{F_p} = \vec{ma}$

$\vec{F_p} = \vec{F_{тяж}} + \vec{F_{сопр}}$

Получим основные уравнения движения мяча, спроектировав силы на оси $x,y,z$:

$m \frac{dV_x}{dt} = -\frac{1}{2}C_dApVV_x$

$m \frac{dV_y}{dt} = -\frac{1}{2}C_dApVV_y $

$m \frac{dV_z}{dt} = -\frac{1}{2}C_dApVV_z - mg$

Воспользуемся условием, что $V_x = \frac{dx}{dt}$
$V_y = \frac{dy}{dt}$
$V_z = \frac{dz}{dt}$

$V = \sqrt{V_x^2 + V_y^2 + V_z^2}$


и разделим обе части уравнений на m:

$\frac{d^2x}{dt^2} = -\frac{C_dApV}{2m} \frac{dx}{dt}$

$\frac{d^2y}{dt^2} = -\frac{C_dApV}{2m}  \frac{dy}{dt}$

$\frac{d^2z}{dt^2} = -\frac{C_dApV}{2m}  \frac{dz}{dt} - g$

При начальных условиях:

$x(t_0) = x_0, y(t_0) = y_0, z(t_0)=z_0;$


$V_x(t_0) = Vcos(\alpha)sin(\theta)$

$V_y(t_0) = Vcos(\alpha)cos(\theta)$

$V_z(t_0) = Vsin(\alpha)$

Условия попадания мяча в ворота(голевые ситуации):

Предполагаем, что ворота стоят при x = a в середине поля по оси у.
1. координата мяча по оси х = a, то есть координата совпадается с координатой ворот
2. координата мяча по оси у лежит в пределах ширины ворот: (b-w)/2 < y < (b+w)/2
3. координата мяча по оси z лежит в пределах высоты ворот: 0 < z< h/2

## Реализация
### Техническое задание
* При заданных начальных условиях положения мяча, направления удара и скорости мяча в начальный момент времени программа должна уметь определять, является ли заданная ситуация голевой.
* Для определения голевой ситуации должны проверяться условия, что в некоторый момент времени:
  1. координата мяча по оси х = a, то есть координата совпадается с координатой ворот
  2. координата мяча по оси у лежит в пределах ширины ворот: (b-w)/2 < y < (b+w)/2
  3. координата мяча по оси z лежит в пределах высоты ворот: 0 < z< h/2
* Для каждой поставленной задачи программа должна позволять изменять и варьироваь начальные данные так, чтобы определять зависимости изменяемых переменных, скорости, площади части поля, из которой можно совершить голевый удар.
* Программа должна численно решать уравнения движения, поставленные в математическом анализе задачи

## Программная реализация
```python 

```
## Численное решение модели
### Определить зависимость угла удара по вертикали от угла к створу ворот для голевой ситуации
Для определения зависимости угла удара по вертикали от угла удара по горизонтали зафиксируем начальные условия: начальное положени мяча, начальная скорость мяча.

* При $V_0 = 35 м/с, (x_0,y_0,z_0) = (80,15,R)$
![](1_1.png)


![](gr1.png)

* При $V_0 = 35 м/с, (x_0,y_0,z_0) = (80,32,R)$

![](1_2.png)


![](gr2.png)

* При $V_0 = 35 м/с, (x_0,y_0,z_0) = (70,45,R)$
![](1_3.png)


![](gr3.png)


###  Опредлить область на поле, откуда спортсмен попадает в створ ворот
При фиксированном значении начальной скорости $V_0 = 35$ и вариации направления удара получим следующий график, на котором отмчена область поля, из которой игрок может попасть в ворота: 
![](mm2.jpg)
###  Определить зависимость площади попадания от углов удара
Определим зависимость площади, из которой можно попасть в ворота от угла удара по вертикали. Для этого зафиксируем значения угла по горигонтали (-35, 0, 55). Для каждого фиксированного значения угла по горизонтали изоразим график зависимости S от alpha
* theta = -35
![](mm2.jpg)

* theta = 0
![](mm2.jpg)

* theta = 55
![](mm2.jpg)

###  Определить зависимость площади попадания от скорости и углов удара 
Определим зависимость площади попадания от скорости и углов удара. Возьмем диапазон рассматриваемых начальных скоростей, для каждой из них будем варьировать начальные углы удара по вертикали и горизонтали и определять площадь части поля, из которой можно попасть в ворота для данной ситуации. По результатам рассчета построим и продемонстрируем граффик зависимости скорости и площади:
![](mm2.jpg)

## Качественный анализ задачи
t = [м / c] / [м / c^2] = [c]

x = y = z = h = w = [м / c] * [c] = [м]

a = b = [м]

V = [м/с]

