import clr

CSharpCodeProvider = clr.Microsoft.CSharp.CSharpCodeProvider
CompilerParameters = clr.System.CodeDom.Compiler.CompilerParameters

parameters = CompilerParameters()
parameters.TreatWarningsAsErrors = False
parameters.GenerateExecutable = False
parameters.CompilerOptions = "/optimize"
parameters.IncludeDebugInformation = False

if __name__ == "__main__":
    parameters.GenerateInMemory = False
    parameters.OutputAssembly = "toy_model.dll"
else:
    parameters.GenerateInMemory = True

csc = CSharpCodeProvider()

results = csc.CompileAssemblyFromFile(parameters, ["toy_model.cs"])

if results.Errors.Count > 0:
    errors = "\n".join(err.ToString() for err in results.Errors)
    raise Exception(errors)
    
if __name__ != "__main__":
    from CSharp import *