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
3. Мяч движется в воздушной среде с плотностью p = 1,2255 кг/м³, коэффициент сопротивления формы $C_d$ примем постоянным и равным 0,47 (соответствует сфере). Площадь поперечного сечения мяча - площадь окружности с радисом r


![](mm2.jpg)

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





