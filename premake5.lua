-- ========================================================================== --
-- premake5.lua                                                               --
-- ========================================================================== --
-- Generates the Visual Studio solution: premake5 vs2022
--
--   mlp    Shared library with the neural network
--   train  Training and evaluation on the MNIST dataset

OUTPUT_DIR = "%{cfg.platform}/%{cfg.buildcfg}"

workspace "MultiLayerPerception"
   architecture "x64"
   configurations { "Debug", "Release" }
   language "C++"
   cppdialect "C++17"
   staticruntime "off"

   filter "system:windows"
      systemversion "latest"
      defines { "PLATFORM_WINDOWS" }

   filter "configurations:Debug"
      defines "DEBUG"
      runtime "Debug"
      symbols "on"

   filter "configurations:Release"
      defines "NDEBUG"
      runtime "Release"
      optimize "on"

   filter {}

project "mlp"
   kind "SharedLib"
   targetdir ("bin/" .. OUTPUT_DIR .. "/mlp")
   objdir ("obj/" .. OUTPUT_DIR .. "/mlp")
   files { "mlp/include/**.h", "mlp/src/**.cpp" }
   includedirs { "mlp/include" }
   defines { "MLP_EXPORT" }

project "train"
   kind "ConsoleApp"
   targetdir ("bin/" .. OUTPUT_DIR .. "/train")
   objdir ("obj/" .. OUTPUT_DIR .. "/train")
   files { "train/**.hpp", "train/**.cpp" }
   includedirs { "mlp/include" }
   libdirs { "bin/" .. OUTPUT_DIR .. "/mlp" }
   links { "mlp" }
   debugdir "%{wks.location}"   -- data/ and models/ are relative to the repo root
   postbuildcommands {
      "{COPY} bin/" .. OUTPUT_DIR .. "/mlp/mlp.dll %{cfg.targetdir}",
   }
