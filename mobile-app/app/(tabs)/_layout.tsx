// mobile-app/app/(tabs)/_layout.tsx
import { Tabs } from "expo-router";
import { Ionicons } from "@expo/vector-icons";

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: "#111",
          borderTopColor: "#333",
          height: 100,              // ← higher to avoid phone nav buttons
          paddingBottom: 40,       // ← extra safe zone
          paddingTop: 2,
        },
        tabBarActiveTintColor: "#00A8FF",
        tabBarInactiveTintColor: "#888",
      }}
    >
      {/* ------------------------------------------------------------- */}
      {/* 1. HOME TAB */}
      {/* ------------------------------------------------------------- */}
      <Tabs.Screen
        name="home"
        options={{
          title: "Home",
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="home-outline" size={24} color={color} />
          ),
        }}
      />

      {/* ------------------------------------------------------------- */}
      {/* 2. HEAD DETECTION TAB */}
      {/* ------------------------------------------------------------- */}
      <Tabs.Screen
        name="head"
        options={{
          title: "Head Detect",
          tabBarIcon: ({ color }) => (
            <Ionicons name="videocam-outline" size={24} color={color} />
          ),
        }}
      />

      {/* ------------------------------------------------------------- */}
      {/* 3. SIGNATURE DETECTION TAB */}
      {/* ------------------------------------------------------------- */}
      <Tabs.Screen
        name="signature"
        options={{
          title: "Signatures",
          tabBarIcon: ({ color }) => (
            <Ionicons name="document-text-outline" size={24} color={color} />
          ),
        }}
      />

      {/* ------------------------------------------------------------- */}
      {/* 4. COMPARE TAB */}
      {/* ------------------------------------------------------------- */}
      <Tabs.Screen
        name="compare"
        options={{
          title: "Compare",
          tabBarIcon: ({ color }) => (
            <Ionicons name="git-compare-outline" size={24} color={color} />
          ),
        }}
      />

      {/* ------------------------------------------------------------- */}
      {/* 5. LIVE FACE TAB */}
      {/* ------------------------------------------------------------- */}
      <Tabs.Screen
        name="live"
        options={{
          title: "Live Face",
          tabBarIcon: ({ color }) => (
            <Ionicons name="scan-outline" size={24} color={color} />
          ),
        }}
      />

    </Tabs>
  );
}
