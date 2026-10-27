-- Run this in SQL Server Management Studio (or sqlcmd) to create the tables.
-- Then set DATABASE_URL in .env to your SQL Server connection string.

CREATE DATABASE PdmDb;
GO
USE PdmDb;
GO

CREATE TABLE dbo.sensor_readings (
    machine_id           VARCHAR(20)  NOT NULL,
    [timestamp]          DATETIME2    NOT NULL,
    temperature          FLOAT        NOT NULL,
    vibration            FLOAT        NOT NULL,
    pressure             FLOAT        NOT NULL,
    rpm                  FLOAT        NOT NULL,
    [current]            FLOAT        NOT NULL,
    failure              INT          NOT NULL DEFAULT 0,
    fails_within_horizon INT          NOT NULL DEFAULT 0,
    CONSTRAINT PK_sensor_readings PRIMARY KEY (machine_id, [timestamp])
);

CREATE TABLE dbo.predictions (
    id            INT IDENTITY(1,1) PRIMARY KEY,
    machine_id    VARCHAR(20)  NOT NULL,
    predicted_at  VARCHAR(40)  NOT NULL,
    failure_risk  FLOAT        NOT NULL,
    is_anomaly    INT          NOT NULL,
    risk_level    VARCHAR(10)  NOT NULL
);
GO

-- Handy query: machines with the highest average risk in the last day
-- SELECT machine_id, AVG(failure_risk) AS avg_risk, COUNT(*) AS n
-- FROM dbo.predictions
-- WHERE predicted_at >= CONVERT(VARCHAR(40), DATEADD(day,-1,SYSUTCDATETIME()), 126)
-- GROUP BY machine_id ORDER BY avg_risk DESC;
