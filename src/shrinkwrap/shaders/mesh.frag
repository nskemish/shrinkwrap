#version 410 core

in vec3 vWorldPosition;
in vec3 vNormal;

uniform vec3 uBaseColor;
uniform vec3 uCameraPosition;

out vec4 FragColor;


void main()
{
    vec3 normal =
        normalize(vNormal);

    vec3 lightDirection =
        normalize(
            vec3(
                -0.45,
                -0.55,
                -1.0
            )
        );

    float diffuse =
        max(
            dot(
                normal,
                -lightDirection
            ),
            0.0
        );

    float ambient = 0.30;

    //
    // Malo "studio" osvetljenja
    // da mesh ne bude potpuno crn
    // sa suprotne strane.
    //
    vec3 secondaryLight =
        normalize(
            vec3(
                0.6,
                0.25,
                -0.4
            )
        );

    float secondary =
        max(
            dot(
                normal,
                -secondaryLight
            ),
            0.0
        );

    float lighting =
        ambient +
        diffuse * 0.60 +
        secondary * 0.18;

    vec3 color =
        uBaseColor *
        lighting;

    //
    // Jednostavan Fresnel highlight.
    //
    vec3 viewDirection =
        normalize(
            uCameraPosition -
            vWorldPosition
        );

    float fresnel =
        pow(
            1.0 -
            max(
                dot(
                    viewDirection,
                    normal
                ),
                0.0
            ),
            3.0
        );

    color +=
        fresnel *
        vec3(0.08);

    FragColor =
        vec4(
            color,
            1.0
        );
}