using System;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;

internal static class DashboardAskPass
{
    private static bool IsPasswordPrompt(string prompt)
    {
        return prompt.IndexOf("password", StringComparison.OrdinalIgnoreCase) >= 0
            || prompt.IndexOf("密码", StringComparison.Ordinal) >= 0;
    }

    [STAThread]
    private static int Main(string[] args)
    {
        byte[] encrypted = null;
        byte[] clear = null;

        try
        {
            string prompt = args.Length > 0 ? args[0] : string.Empty;
            if (!IsPasswordPrompt(prompt))
            {
                return 1;
            }

            string executablePath = Assembly.GetExecutingAssembly().Location;
            string credentialPath = Path.Combine(
                Path.GetDirectoryName(executablePath),
                "credential.bin"
            );

            encrypted = File.ReadAllBytes(credentialPath);
            clear = ProtectedData.Unprotect(
                encrypted,
                null,
                DataProtectionScope.CurrentUser
            );

            Stream standardOutput = Console.OpenStandardOutput();
            standardOutput.Write(clear, 0, clear.Length);
            standardOutput.Flush();
            return 0;
        }
        catch
        {
            // ASKPASS must never print credential or exception details to stderr.
            return 1;
        }
        finally
        {
            if (clear != null)
            {
                Array.Clear(clear, 0, clear.Length);
            }

            if (encrypted != null)
            {
                Array.Clear(encrypted, 0, encrypted.Length);
            }
        }
    }
}
