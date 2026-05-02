Shader "Hidden/Moderato/AI/Tracking/RotatedRoiBlit"
{
    SubShader
    {
        Cull Off
        ZWrite Off
        ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _Roi; // centerX, centerY, width, height in detector top-origin pixels
            float _InputSize;
            float _RoiRotation;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float2 local;
                local.x = (i.uv.x - 0.5) * _Roi.z;
                local.y = (0.5 - i.uv.y) * _Roi.w;

                float s = sin(_RoiRotation);
                float c = cos(_RoiRotation);
                float2 rotated;
                rotated.x = local.x * c - local.y * s;
                rotated.y = -(local.x * s + local.y * c);

                float2 pixel = _Roi.xy + rotated;
                float2 sampleUv = float2(pixel.x / _InputSize, 1.0 - pixel.y / _InputSize);
                sampleUv = clamp(sampleUv, 0.0, 1.0);
                return tex2D(_MainTex, sampleUv);
            }
            ENDCG
        }
    }

    Fallback Off
}
