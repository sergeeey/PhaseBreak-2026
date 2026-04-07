import xgboostlss
import pkgutil

for importer, modname, ispkg in pkgutil.walk_packages(path=xgboostlss.__path__, prefix=xgboostlss.__name__+'.'):
    print(modname, 'PKG' if ispkg else 'MOD')
