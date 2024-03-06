import ninpol
import meshio
i = ninpol.Interpolator()
msh = meshio.read('tests/test-mesh/test1.msh')
m = i.process_mesh(msh)
print(m)