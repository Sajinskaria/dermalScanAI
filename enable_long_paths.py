# Enable Windows Long Paths Registry Setting
# Run this script as Administrator

try:
    import winreg
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                         r"SYSTEM\CurrentControlSet\Control\FileSystem",
                         0, winreg.KEY_SET_VALUE)
    winreg.SetValueEx(key, "LongPathsEnabled", 0, winreg.REG_DWORD, 1)
    winreg.CloseKey(key)
    print("✅ Successfully enabled Windows Long Paths!")
    print("⚠️  Please RESTART your computer for changes to take effect.")
except PermissionError:
    print("❌ ERROR: This script must be run as Administrator!")
    print("\nTo run as Administrator:")
    print("1. Right-click Command Prompt")
    print("2. Select 'Run as Administrator'")
    print("3. Run: python enable_long_paths.py")
except Exception as e:
    print(f"❌ ERROR: {e}")
