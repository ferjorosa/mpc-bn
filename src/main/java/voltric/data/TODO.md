~~Hay que modificar la creación tanto de instancias como de variables para que se les asigne el objeto Data. Para ello,
lo primero es crear un objeto Data(String name) y luego en la creación de cada una de las instancias, se les pasa la referencia del dataset~~

Otra opcion es que tanto Data como DataInstance tengan referencia a las variables para poder
hacer proyecciones individuales sin que salte la excepción de que la instancia no ha sido asignada
a un DataSet.